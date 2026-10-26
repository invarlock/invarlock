//! Closed stored-report checks for presentation; these do not replay provider runs.
use super::presentation::t;
use crate::{EvidenceError, Result};
use serde_json::{json, Value};
fn err(s: impl Into<String>) -> EvidenceError {
    EvidenceError::contract(s)
}
fn fields(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|o| o.len() == keys.len() && keys.iter().all(|k| o.contains_key(*k)))
}
fn number(v: &Value, field: &str) -> Result<f64> {
    if !v.is_number() {
        return Err(err(format!("canonical report {field} must be a number")));
    }
    v.as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| err(format!("canonical report {field} must be finite")))
}
fn integer(v: &Value, field: &str) -> Result<u64> {
    v.as_u64().ok_or_else(|| {
        err(format!(
            "canonical report {field} must be a non-negative integer"
        ))
    })
}
fn close(a: f64, b: f64, rel: f64) -> bool {
    a == b || (a - b).abs() <= 1e-12f64.max(rel * a.abs().max(b.abs()))
}
fn paired(v: &Value, count: u64, value: f64, u: &Value) -> Result<()> {
    if !fields(
        v,
        &[
            "baseline_pass_subject_fail",
            "baseline_fail_subject_pass",
            "both_pass",
            "both_fail",
            "discordant_pairs",
            "mcnemar_exact_two_sided_p_value",
            "effect_size_pp",
            "effect_size_confidence_interval",
        ],
    ) {
        return Err(err("canonical report paired_binary is invalid"));
    }
    let counts = [
        "baseline_pass_subject_fail",
        "baseline_fail_subject_pass",
        "both_pass",
        "both_fail",
        "discordant_pairs",
    ]
    .map(|k| integer(&v[k], &format!("paired_binary.{k}")));
    let [reg, imp, both, fail, discordant] = counts;
    let (reg, imp, both, fail, discordant) = (reg?, imp?, both?, fail?, discordant?);
    if reg as u128 + imp as u128 != discordant as u128
        || reg as u128 + imp as u128 + both as u128 + fail as u128 != count as u128
    {
        return Err(err("canonical report paired_binary counts are invalid"));
    }
    if count > 10_000_000 {
        return Err(EvidenceError::incomplete(
            "local report replay work budget exceeded: paired_binary reconstruction is limited to 10000000 pairs",
        ));
    }
    let mut baseline = vec![true; both as usize + reg as usize];
    baseline.extend(std::iter::repeat_n(false, imp as usize + fail as usize));
    let mut subject = vec![true; both as usize];
    subject.extend(std::iter::repeat_n(false, reg as usize));
    subject.extend(std::iter::repeat_n(true, imp as usize));
    subject.extend(std::iter::repeat_n(false, fail as usize));
    let replay = invarlock_native_core::paired_exact_match_statistics(
        &json!(baseline),
        &json!(subject),
        t(u, "method"),
    )
    .map_err(|e| {
        err(format!(
            "canonical report paired_binary cannot be replayed: {e}"
        ))
    })?;
    let p = number(
        &v["mcnemar_exact_two_sided_p_value"],
        "paired_binary.mcnemar_exact_two_sided_p_value",
    )?;
    let effect = number(&v["effect_size_pp"], "paired_binary.effect_size_pp")?;
    let interval = &v["effect_size_confidence_interval"];
    let expected = &replay["effect_size_confidence_interval"];
    let mut valid = (0.0..=1.0).contains(&p)
        && close(
            p,
            replay["mcnemar_exact_two_sided_p_value"].as_f64().unwrap(),
            1e-12,
        )
        && close(effect, replay["effect_size_pp"].as_f64().unwrap(), 1e-9)
        && close(effect, value, 1e-9)
        && fields(
            interval,
            &["method", "confidence_level", "lower_pp", "upper_pp"],
        )
        && interval["method"] == expected["method"]
        && interval["confidence_level"] == expected["confidence_level"];
    if valid {
        for (k, other) in [("lower_pp", "lower"), ("upper_pp", "upper")] {
            valid &= close(
                number(&interval[k], &format!("paired_binary.interval.{k}"))?,
                expected[k].as_f64().unwrap(),
                1e-9,
            ) && close(
                expected[k].as_f64().unwrap(),
                number(&u[other], &format!("uncertainty.{other}"))?,
                1e-9,
            )
        }
    }
    if !valid {
        return Err(err("canonical report paired_binary values are invalid"));
    }
    Ok(())
}
fn sample(v: &Value, metric: &str, count: u64, lower: f64, upper: f64) -> Result<bool> {
    if !fields(v, &["record_count", "interval_width", "passed"]) {
        return Err(err("canonical report sample_qualification is invalid"));
    }
    let c = &v["record_count"];
    let w = &v["interval_width"];
    if !fields(c, &["minimum", "observed", "passed"]) {
        return Err(err(
            "canonical report sample_qualification record_count is invalid",
        ));
    }
    if !fields(w, &["maximum", "observed", "unit", "passed"]) {
        return Err(err(
            "canonical report sample_qualification interval_width is invalid",
        ));
    }
    let min = integer(&c["minimum"], "sample_qualification.record_count.minimum")?;
    let observed = integer(&c["observed"], "sample_qualification.record_count.observed")?;
    let max = number(&w["maximum"], "sample_qualification.interval_width.maximum")?;
    let width = number(
        &w["observed"],
        "sample_qualification.interval_width.observed",
    )?;
    let unit = if metric == "normalized_nll_per_utf8_byte" {
        "ratio"
    } else {
        "percentage_points"
    };
    if min == 0
        || min > 10000
        || observed != count
        || max <= 0.0
        || (unit == "percentage_points" && max > 200.0)
        || width < 0.0
        || w["unit"] != unit
        || !close(width, upper - lower, 1e-12)
    {
        return Err(err(
            "canonical report sample_qualification values are invalid",
        ));
    }
    let passed = observed >= min && width <= max;
    if c["passed"] != json!(observed >= min)
        || w["passed"] != json!(width <= max)
        || v["passed"] != json!(passed)
    {
        return Err(err(
            "canonical report sample_qualification verdict is invalid",
        ));
    }
    Ok(passed)
}
fn accuracy(v: &Value, metric: &str, means: [f64; 2]) -> Result<bool> {
    if metric != "exact_match" || !fields(v, &["minimum", "baseline", "subject", "passed"]) {
        return Err(err("canonical report side_accuracy is invalid"));
    }
    let min = number(&v["minimum"], "side_accuracy.minimum")?;
    if !(0.0..=1.0).contains(&min) {
        return Err(err("canonical report side_accuracy minimum is invalid"));
    }
    let mut qualified = true;
    for (side, mean) in ["baseline", "subject"].into_iter().zip(means) {
        let q = &v[side];
        if !fields(q, &["observed", "passed"]) {
            return Err(err(format!(
                "canonical report side_accuracy {side} is invalid"
            )));
        }
        let observed = number(&q["observed"], &format!("side_accuracy.{side}.observed"))?;
        if !close(observed, mean, 1e-12) || q["passed"] != json!(mean >= min) {
            return Err(err(format!(
                "canonical report side_accuracy {side} values are invalid"
            )));
        }
        qualified &= mean >= min;
    }
    if v["passed"] != json!(qualified) {
        return Err(err("canonical report side_accuracy verdict is invalid"));
    }
    Ok(qualified)
}
pub(super) fn validate(r: &Value) -> Result<()> {
    let format = t(r, "format");
    if ![
        "invarlock/comparison-report-v1",
        "invarlock/comparison-report-v2",
        "invarlock/comparison-report-v3",
    ]
    .contains(&format)
    {
        return Err(err("canonical comparison report format is invalid"));
    }
    let metric = t(r, "metric");
    let mut keys = vec![
        "format",
        "comparison_id",
        "metric",
        "record_count",
        "baseline",
        "subject",
        "comparison",
        "uncertainty",
        "policy_digest",
        "verdict",
    ];
    match metric {
        "exact_match" => keys.push("paired_binary"),
        "normalized_nll_per_utf8_byte" => keys.push("derived_measurements"),
        _ => keys.extend(["scorer_extension", "scorer_replay"]),
    }
    if r.get("sample_qualification").is_some() {
        keys.push("sample_qualification")
    }
    if r.get("side_accuracy").is_some() {
        if format != "invarlock/comparison-report-v3" {
            return Err(err(
                "canonical report side_accuracy requires comparison-report-v3",
            ));
        }
        keys.push("side_accuracy")
    }
    if !fields(r, &keys) {
        return Err(err("canonical comparison report fields are invalid"));
    }
    for key in ["comparison_id", "metric", "policy_digest"] {
        if t(r, key).is_empty() {
            return Err(err(format!("canonical report {key} is invalid")));
        }
    }
    let count = r["record_count"]
        .as_u64()
        .filter(|v| *v > 0)
        .ok_or_else(|| err("canonical report record_count is invalid"))?;
    let mut means = [0.0; 2];
    for (i, side) in ["baseline", "subject"].iter().enumerate() {
        if !fields(&r[side], &["mean_score"]) {
            return Err(err(format!("canonical report {side} is invalid")));
        }
        means[i] = number(&r[side]["mean_score"], &format!("{side}.mean_score"))?;
    }
    let c = &r["comparison"];
    let kind = t(c, "kind");
    let limit_key = if ["exact_match_delta_pp", "scorer_extension_delta_pp"].contains(&kind) {
        "minimum"
    } else {
        "maximum"
    };
    if ![
        "exact_match_delta_pp",
        "scorer_extension_delta_pp",
        "normalized_nll_ratio",
    ]
    .contains(&kind)
        || !fields(c, &["kind", "value", limit_key])
    {
        return Err(err("canonical report comparison is invalid"));
    }
    let value = number(&c["value"], "comparison.value")?;
    let limit = number(&c[limit_key], &format!("comparison.{limit_key}"))?;
    let expected_kind = match metric {
        "exact_match" => "exact_match_delta_pp",
        "normalized_nll_per_utf8_byte" => "normalized_nll_ratio",
        _ => {
            invarlock_native_core::validate_scorer_binding(&r["scorer_extension"])
                .map_err(|e| err(e.to_string()))?;
            if r["scorer_extension"]["scorer_id"] != metric
                || !fields(&r["scorer_replay"], &["baseline", "subject"])
            {
                return Err(err("canonical report scorer binding is invalid"));
            }
            "scorer_extension_delta_pp"
        }
    };
    if kind != expected_kind {
        return Err(err(
            "canonical report metric and comparison kind do not agree",
        ));
    }
    if !["pass", "fail"].contains(&t(r, "verdict")) {
        return Err(err("canonical report verdict is invalid"));
    }
    let u = &r["uncertainty"];
    let mut keys = vec!["method", "scope", "interval_mass", "lower", "upper"];
    if metric != "exact_match" {
        keys.push("replicates")
    }
    if !fields(u, &keys) {
        return Err(err("canonical report uncertainty is invalid"));
    }
    let method = if metric == "exact_match" {
        if format == "invarlock/comparison-report-v1" {
            "newcombe_hybrid_score_paired_v1"
        } else {
            "newcombe_hybrid_score_paired_v2"
        }
    } else {
        "paired_percentile_bootstrap_sha256_v1"
    };
    if u["method"] != method {
        return Err(err("canonical report uncertainty method is invalid"));
    }
    if u["scope"]
        != if metric == "exact_match" {
            "paired_binary_outcomes"
        } else {
            "authenticated_schedule"
        }
    {
        return Err(err("canonical report uncertainty scope is invalid"));
    }
    if metric != "exact_match" && u["replicates"].as_f64() != Some(2048.0) {
        return Err(err("canonical report uncertainty replicates are invalid"));
    }
    if u["interval_mass"].as_f64() != Some(0.95) {
        return Err(err("canonical report uncertainty mass is invalid"));
    }
    let lower = number(&u["lower"], "uncertainty.lower")?;
    let upper = number(&u["upper"], "uncertainty.upper")?;
    if lower > upper {
        return Err(err("canonical report uncertainty bounds are invalid"));
    }
    let (mut passed, expected) = if kind != "normalized_nll_ratio" {
        if means.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(err(
                "canonical report score means must be between zero and one",
            ));
        }
        if !(-100.0..=100.0).contains(&value) {
            return Err(err("canonical report score comparison is out of range"));
        }
        if !(-100.0..=100.0).contains(&limit) {
            return Err(err("canonical report score policy limit is out of range"));
        }
        if lower < -100.0 || upper > 100.0 {
            return Err(err("canonical report score uncertainty is out of range"));
        }
        let expected = (means[1] - means[0]) * 100.0;
        if !close(value, expected, 1e-12) {
            return Err(err(
                "canonical report comparison value does not match the side means",
            ));
        }
        if metric == "exact_match" {
            paired(&r["paired_binary"], count, value, u)?;
            for (side, discordant, mean) in [
                ("baseline", "baseline_pass_subject_fail", means[0]),
                ("subject", "baseline_fail_subject_pass", means[1]),
            ] {
                let _ = side;
                let expected = (r["paired_binary"]["both_pass"].as_u64().unwrap()
                    + r["paired_binary"][discordant].as_u64().unwrap())
                    as f64
                    / count as f64;
                if !close(mean, expected, 1e-12) {
                    return Err(err(
                        "canonical report side means do not match paired outcome counts",
                    ));
                }
            }
        }
        (lower >= limit, expected)
    } else {
        if means[0] <= 0.0 || means[1] < 0.0 {
            return Err(err(
                "canonical report ratio means must be non-negative with a positive baseline",
            ));
        }
        if value < 0.0 || limit <= 0.0 || lower < 0.0 {
            return Err(err(
                "canonical report ratio comparison and uncertainty are invalid",
            ));
        }
        invarlock_native_core::validated_derived_measurements(&r["derived_measurements"]).map_err(
            |e| {
                err(format!(
                    "canonical report derived measurements are invalid: {e}"
                ))
            },
        )?;
        (upper <= limit, means[1] / means[0])
    };
    if !close(value, expected, 1e-12) {
        return Err(err(
            "canonical report comparison value does not match the side means",
        ));
    }
    if let Some(s) = r.get("sample_qualification") {
        passed &= sample(s, metric, count, lower, upper)?
    }
    if let Some(a) = r.get("side_accuracy") {
        passed &= accuracy(a, metric, means)?
    }
    if r["verdict"] != if passed { "pass" } else { "fail" } {
        return Err(err(
            "canonical report verdict does not match the uncertainty bound and policy limit",
        ));
    }
    Ok(())
}
