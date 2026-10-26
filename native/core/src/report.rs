//! Comparison report replay from verifier-derived native paired records.
use crate::{fsum, paired_exact_match_statistics, paired_resampling_interval, AnalysisError};
use invarlock_native_contracts::{canonical_json_no_newline, sha256_digest};
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn fields(v: &Value, expected: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|o| o.len() == expected.len() && expected.iter().all(|k| o.contains_key(*k)))
}
fn number(v: &Value, label: &str) -> Result<f64, AnalysisError> {
    v.as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| err(format!("{label} must be a finite number")))
}
fn hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub fn validate_scorer_binding(v: &Value) -> Result<(), AnalysisError> {
    if !v.is_object() {
        return Err(err("scorer extension binding must be a JSON object"));
    }
    if !fields(
        v,
        &[
            "format_version",
            "scorer_abi",
            "scorer_id",
            "scorer_version",
            "descriptor_sha256",
            "configuration",
            "configuration_sha256",
        ],
    ) {
        return Err(err("scorer extension binding fields are invalid"));
    }
    if v["format_version"] != "invarlock/scorer-extension-binding-v1" {
        return Err(err("scorer extension binding format is invalid"));
    }
    if v["scorer_abi"] != "1" {
        return Err(err("scorer extension binding ABI is invalid"));
    }
    if !v["configuration"].is_object() {
        return Err(err("scorer extension configuration must be an object"));
    }
    let id = v["scorer_id"].as_str().unwrap_or("");
    if id.split('.').count() < 2
        || !id.split('.').all(|s| {
            !s.is_empty()
                && s.len() <= 64
                && s.as_bytes()[0].is_ascii_lowercase()
                && s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        })
    {
        return Err(err(
            "scorer_id must be a dotted canonical scorer identifier",
        ));
    }
    let version = v["scorer_version"].as_str().unwrap_or("");
    if version.split('.').count() != 3
        || !version.split('.').all(|s| {
            !s.is_empty()
                && s.bytes().all(|b| b.is_ascii_digit())
                && (s == "0" || !s.starts_with('0'))
        })
    {
        return Err(err("scorer_version must be a stable major.minor.patch"));
    }
    for k in ["descriptor_sha256", "configuration_sha256"] {
        if !v[k].as_str().is_some_and(hash) {
            return Err(err(format!("{k} must be a lowercase sha256 digest")));
        }
    }
    let bytes = canonical_json_no_newline(&v["configuration"]).map_err(|e| err(e.to_string()))?;
    if bytes.len() > 65536 {
        return Err(err("configuration exceeds the 65536-byte size limit"));
    }
    if sha256_digest(&bytes).trim_start_matches("sha256:")
        != v["configuration_sha256"].as_str().unwrap()
    {
        return Err(err(
            "configuration_sha256 does not match canonical configuration",
        ));
    }
    Ok(())
}
fn selected<'a>(
    policy: &'a Value,
    metric: &str,
    extension: Option<&Value>,
) -> Result<&'a Value, AnalysisError> {
    if !fields(policy, &["resolved_policy"]) {
        return Err(err("policy fields must contain exactly resolved_policy"));
    }
    let resolved = &policy["resolved_policy"];
    if !fields(resolved, &["metrics"]) {
        return Err(err(
            "policy resolved_policy fields must contain exactly metrics",
        ));
    }
    let metrics = &resolved["metrics"];
    if !metrics.is_object() {
        return Err(err("policy resolved_policy.metrics must be an object"));
    }
    let name = if extension.is_some() {
        "scorer_extension"
    } else {
        metric
    };
    let threshold = if metric == "normalized_nll_per_utf8_byte" {
        "ratio_max"
    } else {
        "delta_min_pp"
    };
    if !fields(metrics, &[name]) {
        return Err(err(format!(
            "policy metrics must contain exactly metrics.{name}"
        )));
    }
    let value = &metrics[name];
    let obj = value
        .as_object()
        .ok_or_else(|| err(format!("policy metrics.{name} must be an object")))?;
    let width = if metric == "normalized_nll_per_utf8_byte" {
        "maximum_interval_width_ratio"
    } else {
        "maximum_interval_width_pp"
    };
    let has_min = obj.contains_key("minimum_record_count");
    let has_width = obj.contains_key(width);
    if has_min != has_width {
        return Err(err(format!(
            "policy metrics.{name}.minimum_record_count and {width} must be provided together"
        )));
    }
    let mut keys = BTreeSet::from([threshold]);
    if let Some(binding) = extension {
        for k in [
            "scorer_id",
            "scorer_version",
            "descriptor_sha256",
            "configuration_sha256",
        ] {
            keys.insert(k);
            if value[k] != binding[k] {
                return Err(err(
                    "policy scorer_extension pin does not match the request binding",
                ));
            }
        }
    }
    if has_min {
        keys.insert("minimum_record_count");
        keys.insert(width);
    }
    if obj.contains_key("minimum_side_accuracy") {
        if metric != "exact_match" || extension.is_some() {
            return Err(err(format!(
                "policy metrics.{name}.minimum_side_accuracy is only supported for exact_match"
            )));
        }
        keys.insert("minimum_side_accuracy");
    }
    if obj.len() != keys.len() || obj.keys().any(|k| !keys.contains(k.as_str())) {
        return Err(err(format!(
            "policy metrics.{name} must contain exactly the authorized fields"
        )));
    }
    if has_min {
        if !value["minimum_record_count"]
            .as_u64()
            .is_some_and(|n| (1..=50000).contains(&n))
        {
            return Err(err(format!(
                "policy metrics.{name}.minimum_record_count must be an integer between 1 and 50000"
            )));
        }
        let maximum = number(&value[width], &format!("policy metrics.{name}.{width}"))?;
        if maximum <= 0.0 || width.ends_with("_pp") && maximum > 200.0 {
            return Err(err(format!(
                "policy metrics.{name}.{width} must be positive{}",
                if width.ends_with("_pp") {
                    " and at most 200"
                } else {
                    ""
                }
            )));
        }
    }
    if obj.contains_key("minimum_side_accuracy") {
        let minimum = number(
            &value["minimum_side_accuracy"],
            &format!("policy metrics.{name}.minimum_side_accuracy"),
        )?;
        if !(0.0..=1.0).contains(&minimum) {
            return Err(err(format!(
                "policy metrics.{name}.minimum_side_accuracy must be between 0 and 1"
            )));
        }
    }
    Ok(value)
}
pub fn validated_derived_measurements(value: &Value) -> Result<Value, AnalysisError> {
    if !fields(value, &["perplexity_ratio"]) {
        return Err(err(
            "normalized NLL derived_measurements must contain exactly perplexity_ratio",
        ));
    }
    let m = &value["perplexity_ratio"];
    if !m.is_object() {
        return Err(err("derived perplexity_ratio must be an object"));
    }
    if m["basis"] != "authenticated_target_likelihood" {
        return Err(err("derived perplexity_ratio basis is invalid"));
    }
    if m["method"] != "target_token_weighted_perplexity_ratio_v1" {
        return Err(err("derived perplexity_ratio method is invalid"));
    }
    if m["status"] == "unavailable" {
        if !fields(m, &["status", "basis", "method", "reason"])
            || !m["reason"].as_str().is_some_and(|s| {
                [
                    "tokenizer_contracts_differ",
                    "target_token_counts_unavailable",
                    "target_token_counts_differ",
                    "derived_value_non_finite",
                ]
                .contains(&s)
            })
        {
            return Err(err("derived perplexity_ratio unavailability is invalid"));
        }
        return Ok(value.clone());
    }
    if m["status"] != "available"
        || !fields(
            m,
            &[
                "status",
                "basis",
                "method",
                "tokenizer_metadata_sha256",
                "target_token_count",
                "baseline_perplexity",
                "subject_perplexity",
                "ratio",
            ],
        )
    {
        return Err(err("derived perplexity_ratio fields are invalid"));
    }
    if !m["tokenizer_metadata_sha256"].as_str().is_some_and(hash)
        || m["target_token_count"].as_u64().is_none_or(|n| n == 0)
    {
        return Err(err("derived perplexity_ratio bindings are invalid"));
    }
    let b = number(&m["baseline_perplexity"], "derived baseline_perplexity")?;
    let s = number(&m["subject_perplexity"], "derived subject_perplexity")?;
    let r = number(&m["ratio"], "derived perplexity_ratio")?;
    let quotient = s / b;
    let difference = (r - quotient).abs();
    if b.min(s).min(r) <= 0.0
        || !(r == quotient
            || quotient.is_finite()
                && difference <= (1e-12 * r).abs().max((1e-12 * quotient).abs()))
    {
        return Err(err("derived perplexity_ratio values are invalid"));
    }
    Ok(value.clone())
}
pub fn build_comparison_report(
    comparison_id: &str,
    paired_records: &Value,
    policy: &Value,
    policy_digest: &str,
    report_format: &str,
) -> Result<Value, AnalysisError> {
    if comparison_id.is_empty()
        || comparison_id.len() > 128
        || !comparison_id.as_bytes()[0].is_ascii_alphanumeric()
        || !comparison_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
    {
        return Err(err("comparison_id is invalid"));
    }
    if ![
        "invarlock/comparison-report-v1",
        "invarlock/comparison-report-v2",
        "invarlock/comparison-report-v3",
    ]
    .contains(&report_format)
    {
        return Err(err("comparison report format is unsupported"));
    }
    let metric = paired_records["metric"].as_str().unwrap_or("");
    let likelihood = metric == "normalized_nll_per_utf8_byte";
    let extension = if !["exact_match", "normalized_nll_per_utf8_byte"].contains(&metric)
        && paired_records.get("scorer_extension").is_some()
        && paired_records.get("scorer_replay").is_some()
    {
        let b = &paired_records["scorer_extension"];
        validate_scorer_binding(b)?;
        if b["scorer_id"] != metric {
            return Err(err("paired records scorer ID is invalid"));
        }
        let replay = &paired_records["scorer_replay"];
        if !fields(replay, &["baseline", "subject"])
            || !replay["baseline"].is_object()
            || !replay["subject"].is_object()
        {
            return Err(err("paired records scorer replay is invalid"));
        }
        Some(b)
    } else {
        None
    };
    let mut keys = vec!["format", "metric", "schedule_sha256", "records"];
    if likelihood {
        keys.push("derived_measurements")
    }
    if extension.is_some() {
        keys.extend(["scorer_extension", "scorer_replay"])
    }
    if !fields(paired_records, &keys) {
        return Err(err("paired records fields are invalid"));
    }
    if paired_records["format"] != "invarlock/paired-records-v1" {
        return Err(err("paired records format is invalid"));
    }
    if !["exact_match", "normalized_nll_per_utf8_byte"].contains(&metric) && extension.is_none() {
        return Err(err("paired records metric is unsupported"));
    }
    let records = paired_records["records"]
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| err("paired records must be a non-empty array"))?;
    let mut b = Vec::new();
    let mut s = Vec::new();
    for (i, r) in records.iter().enumerate() {
        if !r.is_object() {
            return Err(err(format!("paired record {i} must be an object")));
        }
        for (side, target) in [("baseline", &mut b), ("subject", &mut s)] {
            if !r[side].is_object() {
                return Err(err(format!("paired record {i} {side} is invalid")));
            }
            target.push(number(
                &r[side]["score"],
                &format!("record {i} {side}.score"),
            )?)
        }
        if r.get("weight").is_some() {
            return Err(err(format!("paired record {i} weight is invalid")));
        }
    }
    let n = b.len() as f64;
    let bm = fsum(b.iter().copied()).map_err(|e| err(e.to_string()))? / n;
    let sm = fsum(s.iter().copied()).map_err(|e| err(e.to_string()))? / n;
    let value = if likelihood {
        if bm <= 0.0 {
            return Err(err(
                "normalized NLL baseline mean must be greater than zero",
            ));
        }
        sm / bm
    } else {
        fsum(b.iter().zip(&s).map(|(b, s)| s - b)).map_err(|e| err(e.to_string()))? / n * 100.0
    };
    let seed = paired_records["schedule_sha256"]
        .as_str()
        .ok_or_else(|| err("paired records schedule_sha256 is invalid"))?;
    let selected = selected(policy, metric, extension)?;
    if selected.get("minimum_side_accuracy").is_some()
        && report_format != "invarlock/comparison-report-v3"
    {
        return Err(err(
            "minimum_side_accuracy requires invarlock/comparison-report-v3",
        ));
    }
    let mut paired_binary = None;
    let (lower, upper, uncertainty, comparison, mut passed) = if metric == "exact_match" {
        let method = if report_format == "invarlock/comparison-report-v1" {
            "newcombe_hybrid_score_paired_v1"
        } else {
            "newcombe_hybrid_score_paired_v2"
        };
        let stats = paired_exact_match_statistics(&json!(b), &json!(s), method)
            .map_err(|e| err(format!("exact-match paired statistics are invalid: {e}")))?;
        let ci = &stats["effect_size_confidence_interval"];
        let lower = ci["lower_pp"].as_f64().unwrap();
        let upper = ci["upper_pp"].as_f64().unwrap();
        paired_binary = Some(
            json!({"baseline_pass_subject_fail":stats["baseline_pass_subject_fail_count"],"baseline_fail_subject_pass":stats["baseline_fail_subject_pass_count"],"both_pass":stats["both_pass_count"],"both_fail":stats["both_fail_count"],"discordant_pairs":stats["baseline_pass_subject_fail_count"].as_u64().unwrap()+stats["baseline_fail_subject_pass_count"].as_u64().unwrap(),"mcnemar_exact_two_sided_p_value":stats["mcnemar_exact_two_sided_p_value"],"effect_size_pp":stats["effect_size_pp"],"effect_size_confidence_interval":ci}),
        );
        let limit = number(&selected["delta_min_pp"], "policy exact_match.delta_min_pp")?;
        if !(-100.0..=100.0).contains(&limit) {
            return Err(err(
                "policy exact_match.delta_min_pp must be between -100 and 100",
            ));
        }
        (
            lower,
            upper,
            json!({"method":method,"scope":"paired_binary_outcomes","interval_mass":0.95,"lower":lower,"upper":upper}),
            json!({"kind":"exact_match_delta_pp","value":if value==0.0{0.0}else{value},"minimum":limit}),
            lower >= limit,
        )
    } else {
        let (lower, upper) =
            paired_resampling_interval(metric, &b, &s, seed).map_err(|e| err(e.to_string()))?;
        let uncertainty = json!({"method":"paired_percentile_bootstrap_sha256_v1","scope":"authenticated_schedule","interval_mass":0.95,"replicates":2048,"lower":lower,"upper":upper});
        if likelihood {
            let limit = number(
                &selected["ratio_max"],
                "policy normalized_nll_per_utf8_byte.ratio_max",
            )?;
            if limit <= 0.0 {
                return Err(err(
                    "policy normalized_nll_per_utf8_byte.ratio_max must be positive",
                ));
            }
            (
                lower,
                upper,
                uncertainty,
                json!({"kind":"normalized_nll_ratio","value":value,"maximum":limit}),
                upper <= limit,
            )
        } else {
            let limit = number(
                &selected["delta_min_pp"],
                "policy scorer_extension.delta_min_pp",
            )?;
            if !(-100.0..=100.0).contains(&limit) {
                return Err(err(
                    "policy scorer_extension.delta_min_pp must be between -100 and 100",
                ));
            }
            (
                lower,
                upper,
                uncertainty,
                json!({"kind":"scorer_extension_delta_pp","value":if value==0.0{0.0}else{value},"minimum":limit}),
                lower >= limit,
            )
        }
    };
    let mut sample = None;
    if let Some(minimum) = selected.get("minimum_record_count") {
        let width = if likelihood {
            "maximum_interval_width_ratio"
        } else {
            "maximum_interval_width_pp"
        };
        let maximum = &selected[width];
        let counts = records.len() as u64 >= minimum.as_u64().unwrap();
        let width_pass = upper - lower <= maximum.as_f64().unwrap();
        passed &= counts && width_pass;
        sample = Some(
            json!({"record_count":{"minimum":minimum,"observed":records.len(),"passed":counts},"interval_width":{"maximum":maximum,"observed":upper-lower,"unit":if likelihood{"ratio"}else{"percentage_points"},"passed":width_pass},"passed":counts&&width_pass}),
        )
    }
    let mut side = None;
    if let Some(minimum) = selected.get("minimum_side_accuracy") {
        let limit = minimum.as_f64().unwrap();
        let bp = bm >= limit;
        let sp = sm >= limit;
        passed &= bp && sp;
        side = Some(
            json!({"minimum":minimum,"baseline":{"observed":bm,"passed":bp},"subject":{"observed":sm,"passed":sp},"passed":bp&&sp}),
        )
    }
    let policy_digest = policy_digest.trim().to_lowercase();
    if !policy_digest.strip_prefix("sha256:").is_some_and(hash) {
        return Err(err("policy digest must be a sha256:... digest"));
    }
    let mut report = json!({"format":report_format,"comparison_id":comparison_id,"metric":metric,"record_count":records.len(),"baseline":{"mean_score":bm},"subject":{"mean_score":sm},"comparison":comparison,"uncertainty":uncertainty,"policy_digest":policy_digest,"verdict":if passed{"pass"}else{"fail"}});
    if likelihood {
        report["derived_measurements"] =
            validated_derived_measurements(&paired_records["derived_measurements"])?
    }
    if let Some(binary) = paired_binary {
        report["paired_binary"] = binary
    }
    if let Some(sample) = sample {
        report["sample_qualification"] = sample
    }
    if let Some(side) = side {
        report["side_accuracy"] = side
    }
    if let Some(binding) = extension {
        report["scorer_extension"] = binding.clone();
        report["scorer_replay"] = paired_records["scorer_replay"].clone()
    }
    Ok(report)
}
