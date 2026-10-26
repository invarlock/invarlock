//! Exact rational collapse and directed decimal fixed-benchmark inference.
//!
//! Decimal logarithms are enclosed by fixed-point atanh series with a proved
//! geometric tail bound. Precision increases until both bounds round to the
//! same decimal. Square roots use integer square roots and exact midpoint
//! comparisons. No binary floating-point arithmetic enters these intervals.
use crate::AnalysisError;
use num_bigint::BigInt;
use num_rational::BigRational as Rat;
use num_traits::{One, Signed, Zero};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
const METHOD: &str = "fixed-benchmark-hoeffding-v1";
fn err(s: impl Into<String>) -> AnalysisError {
    AnalysisError::new(s)
}
fn integer(n: usize) -> Rat {
    Rat::from_integer(BigInt::from(n))
}
fn ten(e: i32) -> Rat {
    if e >= 0 {
        Rat::from_integer(BigInt::from(10).pow(e as u32))
    } else {
        Rat::new(BigInt::one(), BigInt::from(10).pow((-e) as u32))
    }
}
fn dec(v: &Value, name: &str) -> Result<Rat, AnalysisError> {
    let s = v
        .as_str()
        .ok_or_else(|| err(format!("{name} must be a finite Decimal")))?;
    crate::scoring::decimal(s).map_err(|_| err(format!("{name} must be a finite Decimal")))
}
fn positive(v: &Value, name: &str) -> Result<usize, AnalysisError> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0)
        .ok_or_else(|| err(format!("{name} must be a positive integer")))
}
fn positive_big(v: &Value, name: &str) -> Result<BigInt, AnalysisError> {
    if !v.is_number() || v.to_string().contains(['.', 'e', 'E']) {
        return Err(err(format!("{name} must be a positive integer")));
    }
    v.to_string()
        .parse::<BigInt>()
        .ok()
        .filter(|n| n.is_positive())
        .ok_or_else(|| err(format!("{name} must be a positive integer")))
}
fn json_integer(n: &BigInt) -> Value {
    serde_json::from_str(&n.to_string()).unwrap()
}
fn adjusted(v: &Rat) -> i32 {
    if v.is_zero() {
        return 0;
    }
    let abs = v.abs();
    let mut e = abs.numer().to_string().len() as i32 - abs.denom().to_string().len() as i32;
    if abs < ten(e) {
        e -= 1
    }
    e
}
#[derive(Clone, Copy)]
enum Rounding {
    Ceiling,
    Floor,
    HalfEven,
}
fn quotient(n: &BigInt, d: &BigInt, rounding: Rounding) -> BigInt {
    let q = n / d;
    let r = n % d;
    if r.is_zero() {
        return q;
    }
    match rounding {
        Rounding::Ceiling => {
            if n.is_positive() {
                q + 1
            } else {
                q
            }
        }
        Rounding::Floor => {
            if n.is_negative() {
                q - 1
            } else {
                q
            }
        }
        Rounding::HalfEven => {
            let twice = r.abs() * 2;
            let increment = twice > *d || twice == *d && (&q % 2u8) != BigInt::zero();
            if increment {
                q + if n.is_negative() { -1 } else { 1 }
            } else {
                q
            }
        }
    }
}
fn quantize(v: &Rat, exponent: i32, rounding: Rounding) -> Rat {
    let unit = ten(exponent);
    let x = v / &unit;
    Rat::from_integer(quotient(x.numer(), x.denom(), rounding)) * unit
}
fn round(v: &Rat, precision: usize, rounding: Rounding) -> Rat {
    quantize(v, adjusted(v) - precision as i32 + 1, rounding)
}
fn next_plus(v: &Rat, precision: usize) -> Rat {
    v + ten(adjusted(v) - precision as i32 + 1)
}
fn sqrt_round(v: &Rat, precision: usize) -> Rat {
    if v.is_zero() {
        return Rat::zero();
    }
    let exponent = adjusted(v).div_euclid(2) - precision as i32 + 1;
    let unit = ten(exponent);
    let scaled = v / (&unit * &unit);
    let floor = (scaled.numer() / scaled.denom()).sqrt();
    let midpoint = Rat::new(&floor * 2 + 1, BigInt::from(2));
    let square = &midpoint * &midpoint;
    let up = scaled > square || scaled == square && (&floor % 2u8) != BigInt::zero();
    Rat::from_integer(if up { floor + 1 } else { floor }) * unit
}
fn ceil_positive(n: BigInt, d: &BigInt) -> BigInt {
    let (q, r) = (&n / d, &n % d);
    if r.is_zero() {
        q
    } else {
        q + 1
    }
}
fn log_series_bounds(x: &Rat, digits: usize) -> (BigInt, BigInt) {
    let scale = BigInt::from(10).pow(digits as u32);
    let z = (x - Rat::one()) / (x + Rat::one());
    let z_scaled = &z * Rat::from_integer(scale.clone());
    let zl = z_scaled.numer() / z_scaled.denom();
    let zu = ceil_positive(z_scaled.numer().clone(), z_scaled.denom());
    let z2l = &zl * &zl / &scale;
    let z2u = ceil_positive(&zu * &zu, &scale);
    let (mut power_l, mut power_u) = (zl, zu);
    let (mut sum_l, mut sum_u) = (BigInt::zero(), BigInt::zero());
    let terms = digits * 2 + 16;
    for k in 0..terms {
        let denominator = BigInt::from(2 * k + 1);
        sum_l += &power_l / &denominator;
        sum_u += ceil_positive(power_u.clone(), &denominator);
        power_l = power_l * &z2l / &scale;
        power_u = ceil_positive(power_u * &z2u, &scale)
    }
    // z <= 1/3, so 2/(1-z²) <= 9/4 < 3 bounds all omitted terms.
    let tail = ceil_positive(power_u * 3, &BigInt::from(2 * terms + 1));
    (sum_l * 2, sum_u * 2 + tail)
}
fn ln_round(v: &Rat, precision: usize) -> Result<Rat, AnalysisError> {
    let mut power = v.numer().bits().saturating_sub(v.denom().bits()) as usize;
    let mut reduced = v / Rat::from_integer(BigInt::one() << power);
    if reduced < Rat::one() && power > 0 {
        reduced *= integer(2);
        power -= 1;
    }
    if reduced >= integer(2) {
        reduced /= integer(2);
        power += 1;
    }
    if reduced < Rat::one() {
        return Err(err("logarithm argument must be at least one"));
    }
    for extra in [20usize, 40, 80, 160] {
        let digits = precision + extra;
        let scale = Rat::from_integer(BigInt::from(10).pow(digits as u32));
        let (lo, hi) = log_series_bounds(&reduced, digits);
        let (l2, h2) = log_series_bounds(&integer(2), digits);
        let lower = Rat::from_integer(lo + l2 * power) / &scale;
        let upper = Rat::from_integer(hi + h2 * power) / &scale;
        let a = round(&lower, precision, Rounding::HalfEven);
        let b = round(&upper, precision, Rounding::HalfEven);
        if a == b {
            return Ok(a);
        }
    }
    Err(err(
        "decimal logarithm rounding could not be resolved within the arithmetic limit",
    ))
}
fn decimal_string(v: &Rat, exponent: i32) -> String {
    let scaled = v / ten(exponent);
    debug_assert!(scaled.is_integer());
    let number = scaled.to_integer();
    let negative = number.is_negative();
    let mut digits = number.abs().to_string();
    let adjusted = digits.len() as i32 + exponent - 1;
    let prefix = if negative { "-" } else { "" };
    if exponent > 0 || adjusted < -6 {
        if digits.len() > 1 {
            digits.insert(1, '.')
        }
        return format!(
            "{prefix}{digits}E{}{adjusted}",
            if adjusted >= 0 { "+" } else { "" }
        );
    }
    if exponent == 0 {
        return format!("{prefix}{digits}");
    }
    let places = (-exponent) as usize;
    if digits.len() <= places {
        digits = format!("{}{}", "0".repeat(places + 1 - digits.len()), digits)
    }
    let point = digits.len() - places;
    digits.insert(point, '.');
    format!("{prefix}{digits}")
}
fn source_decimal_string(raw: &str, value: &Rat) -> String {
    let raw = raw.trim().replace('_', "");
    let mut parts = raw.split(['e', 'E']);
    let mantissa = parts.next().unwrap_or("");
    let exponent = parts
        .next()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0)
        - mantissa.split('.').nth(1).map(str::len).unwrap_or(0) as i32;
    decimal_string(value, exponent)
}

fn fraction(v: &Value) -> Result<Rat, AnalysisError> {
    if v.is_string() {
        return dec(v, "unit value");
    }
    let n = v["numerator"]
        .as_str()
        .and_then(|s| s.parse::<BigInt>().ok());
    let d = v["denominator"]
        .as_str()
        .and_then(|s| s.parse::<BigInt>().ok());
    match (n, d) {
        (Some(n), Some(d)) if !d.is_zero() => Ok(Rat::new(n, d)),
        _ => Err(err("unit values must be finite Decimals or Fractions")),
    }
}
fn fraction_json(v: &Rat) -> Value {
    json!({"numerator":v.numer().to_string(),"denominator":v.denom().to_string()})
}
#[derive(Clone)]
struct Interval {
    mean: Rat,
    lower: Rat,
    upper: Rat,
    count: usize,
    alpha_text: String,
    comparisons: BigInt,
    exponent: i32,
    negative_zero: [bool; 3],
}
impl Interval {
    fn json(&self) -> Value {
        let render = |v: &Rat, index: usize| {
            let text = decimal_string(v, self.exponent);
            if v.is_zero() && self.negative_zero[index] {
                format!("-{text}")
            } else {
                text
            }
        };
        json!({"mean":render(&self.mean,0),"lower":render(&self.lower,1),"upper":render(&self.upper,2),"unit_count":self.count,"alpha":self.alpha_text,"comparisons":json_integer(&self.comparisons),"method":METHOD})
    }
}
fn interval(
    values: &[Rat],
    lower: &Rat,
    upper: &Rat,
    alpha: &Rat,
    alpha_text: &str,
    comparisons: &BigInt,
    exponent: i32,
) -> Result<Interval, AnalysisError> {
    if lower > upper {
        return Err(err("lower_bound must not exceed upper_bound"));
    }
    if alpha <= &Rat::zero() || alpha >= &Rat::one() {
        return Err(err("alpha must be strictly between zero and one"));
    }
    if comparisons <= &BigInt::zero() {
        return Err(err("comparisons must be a positive integer"));
    }
    if values.iter().any(|v| v < lower || v > upper) {
        return Err(err("unit value is outside the declared bounds"));
    }
    if values.is_empty() {
        return Err(err("at least one independent unit is required"));
    }
    let mean = values.iter().cloned().sum::<Rat>() / integer(values.len());
    let precision = 100usize
        .max((adjusted(lower) - exponent + 20).max(0) as usize)
        .max((adjusted(upper) - exponent + 20).max(0) as usize);
    let width = round(&(upper - lower), precision, Rounding::Ceiling);
    let arg = round(
        &(Rat::from_integer(comparisons * 2) / alpha),
        precision,
        Rounding::Ceiling,
    );
    let log = next_plus(&ln_round(&arg, precision)?, precision);
    let quotient = round(
        &(log / integer(2 * values.len())),
        precision,
        Rounding::Ceiling,
    );
    let root = next_plus(&sqrt_round(&quotient, precision), precision);
    let radius = round(&(width * root), precision, Rounding::Ceiling);
    let mean_upper = round(&mean, precision, Rounding::Ceiling);
    let high = round(&(mean_upper + &radius), precision, Rounding::Ceiling).min(upper.clone());
    let high_negative = high.is_negative();
    let high = quantize(&high, exponent, Rounding::Ceiling);
    let mean_lower = round(&mean, precision, Rounding::Floor);
    let low = round(&(mean_lower - &radius), precision, Rounding::Floor).max(lower.clone());
    let low_negative = low.is_negative();
    let low = quantize(&low, exponent, Rounding::Floor);
    let reported_mean = quantize(
        &round(&mean, precision, Rounding::HalfEven),
        exponent,
        Rounding::HalfEven,
    );
    Ok(Interval {
        mean: reported_mean,
        lower: low,
        upper: high,
        count: values.len(),
        alpha_text: source_decimal_string(alpha_text, alpha),
        comparisons: comparisons.clone(),
        exponent,
        negative_zero: [mean.is_negative(), low_negative, high_negative],
    })
}
fn quantum(v: &Value) -> Result<i32, AnalysisError> {
    let raw = v
        .as_str()
        .ok_or_else(|| err("quantum must be a finite Decimal"))?;
    let q = crate::scoring::decimal(raw).map_err(|_| err("quantum must be a finite Decimal"))?;
    let mantissa = raw
        .split(['e', 'E'])
        .next()
        .unwrap()
        .trim_start_matches('+');
    let digits = mantissa.replace('.', "");
    if q <= Rat::zero() || digits.trim_start_matches('0') != "1" {
        return Err(err("quantum must be a positive power of ten"));
    }
    let exponent = adjusted(&q);
    if q != ten(exponent) {
        return Err(err("quantum must be a positive power of ten"));
    }
    Ok(exponent)
}
fn interval_request(v: &Value) -> Result<Interval, AnalysisError> {
    let lower = dec(&v["lower_bound"], "lower_bound")?;
    let upper = dec(&v["upper_bound"], "upper_bound")?;
    let alpha_v = v.get("alpha").cloned().unwrap_or(json!("0.05"));
    let alpha = dec(&alpha_v, "alpha")?;
    let q = v
        .get("quantum")
        .cloned()
        .unwrap_or(json!("0.000000000000001"));
    let exp = quantum(&q)?;
    let comparisons = positive_big(v.get("comparisons").unwrap_or(&json!(1)), "comparisons")?;
    let values = v["values"]
        .as_array()
        .ok_or_else(|| err("unit values must be finite Decimals or Fractions"))?
        .iter()
        .map(fraction)
        .collect::<Result<Vec<_>, _>>()?;
    let mut result = interval(
        &values,
        &lower,
        &upper,
        &alpha,
        alpha_v.as_str().unwrap(),
        &comparisons,
        exp,
    )?;
    if lower.is_zero() && result.lower.is_zero() {
        result.negative_zero[1] = v["lower_bound"].as_str().unwrap().trim().starts_with('-');
    }
    if upper.is_zero() && result.upper.is_zero() {
        result.negative_zero[2] = v["upper_bound"].as_str().unwrap().trim().starts_with('-');
    }
    Ok(result)
}
fn collapse(v: &Value) -> Result<Value, AnalysisError> {
    let repetitions = positive(&v["repetitions"], "repetitions")?;
    let lower = dec(&v["lower_bound"], "lower_bound")?;
    let upper = dec(&v["upper_bound"], "upper_bound")?;
    if lower > upper {
        return Err(err("lower_bound must not exceed upper_bound"));
    }
    let cases = v["cases"]
        .as_array()
        .ok_or_else(|| err("at least one scheduled case is required"))?;
    let mut scheduled = BTreeMap::new();
    for case in cases {
        let id = case["case_id"].as_str().unwrap_or("");
        let unit = case["unit_id"].as_str().unwrap_or("");
        if id.is_empty() || unit.is_empty() || scheduled.insert(id, unit).is_some() {
            return Err(err(
                "scheduled cases need unique case IDs and nonempty unit IDs",
            ));
        }
    }
    if scheduled.is_empty() {
        return Err(err("at least one scheduled case is required"));
    }
    let mut observations = BTreeMap::new();
    for score in v["scores"]
        .as_array()
        .ok_or_else(|| err("missing required score slots"))?
    {
        let id = score["case_id"].as_str().unwrap_or("");
        let side = score["side"].as_str().unwrap_or("");
        let repetition = score["repetition"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok());
        if !scheduled.contains_key(id)
            || !["baseline", "subject"].contains(&side)
            || repetition.is_none_or(|r| r >= repetitions)
        {
            return Err(err("score does not match a scheduled slot"));
        }
        let value = dec(&score["value"], "score")?;
        if value < lower || value > upper {
            return Err(err("score is outside the declared bounds"));
        }
        if observations
            .insert((id, side, repetition.unwrap()), value)
            .is_some()
        {
            return Err(err("duplicate score slot"));
        }
    }
    if observations.len() != scheduled.len() * 2 * repetitions {
        return Err(err("missing required score slots"));
    }
    let mut grouped = BTreeMap::<&str, Vec<(Rat, Rat)>>::new();
    for (&id, &unit) in &scheduled {
        let means = ["baseline", "subject"].map(|side| {
            (0..repetitions)
                .map(|r| observations[&(id, side, r)].clone())
                .sum::<Rat>()
                / integer(repetitions)
        });
        grouped
            .entry(unit)
            .or_default()
            .push((means[0].clone(), means[1].clone()))
    }
    let units = grouped
        .into_iter()
        .map(|(unit, values)| {
            let b = values.iter().map(|(b, _)| b.clone()).sum::<Rat>() / integer(values.len());
            let s = values.iter().map(|(_, s)| s.clone()).sum::<Rat>() / integer(values.len());
            json!({"unit_id":unit,"baseline":fraction_json(&b),"subject":fraction_json(&s)})
        })
        .collect::<Vec<_>>();
    Ok(json!({"units":units,"case_count":scheduled.len(),"repetitions":repetitions}))
}
fn decision(v: &Value, subject_bound: bool) -> Result<Value, AnalysisError> {
    let direction = v["direction"].as_str().unwrap_or("");
    if !["higher", "lower"].contains(&direction) {
        return Err(err("direction must be higher or lower"));
    }
    let field = if subject_bound { "bound" } else { "margin" };
    let threshold = dec(&v[field], field)?;
    let minimum = positive_big(v.get("minimum_units").unwrap_or(&json!(1)), "minimum_units")?;
    if !subject_bound && threshold < Rat::zero() {
        return Err(err("margin must be nonnegative"));
    }
    let interval = &v["interval"];
    let count = interval["unit_count"].as_u64().unwrap_or(0) as usize;
    if BigInt::from(count) < minimum {
        return Ok(json!("insufficient_evidence"));
    }
    let mut lower = dec(&interval["lower"], "lower")?;
    let mut upper = dec(&interval["upper"], "upper")?;
    let value = if subject_bound {
        if direction == "higher" {
            if lower >= threshold {
                "pass"
            } else if upper < threshold {
                "regression"
            } else {
                "insufficient_evidence"
            }
        } else if upper <= threshold {
            "pass"
        } else if lower > threshold {
            "regression"
        } else {
            "insufficient_evidence"
        }
    } else {
        if direction == "lower" {
            (lower, upper) = (-upper, -lower)
        }
        let threshold = -threshold;
        if lower >= threshold {
            "pass"
        } else if upper < threshold {
            "regression"
        } else {
            "insufficient_evidence"
        }
    };
    Ok(json!(value))
}
fn combine(v: &Value) -> Result<Value, AnalysisError> {
    let mut required = BTreeSet::new();
    for gate in v["gates"]
        .as_array()
        .ok_or_else(|| err("unknown gate decision"))?
    {
        let decision = gate["decision"].as_str().unwrap_or("");
        if !["pass", "regression", "insufficient_evidence"].contains(&decision) {
            return Err(err("unknown gate decision"));
        }
        let is_required = gate
            .get("required")
            .unwrap_or(&Value::Bool(true))
            .as_bool()
            .ok_or_else(|| err("required must be boolean"))?;
        if is_required {
            required.insert(decision);
        }
    }
    Ok(json!(if required.contains("regression") {
        "regression"
    } else if required.is_empty() || required.contains("insufficient_evidence") {
        "insufficient_evidence"
    } else {
        "pass"
    }))
}
/// Explicit numerical diagnostics. These inputs do not authenticate a judge
/// execution; retained-measurement provenance is a separate verifier boundary.
pub fn judge_statistics(request: &Value) -> Result<Value, AnalysisError> {
    match request["statistic"].as_str().unwrap_or("") {
        "collapse_scores" => collapse(request),
        "hoeffding_interval" => interval_request(request).map(|v| v.json()),
        "decide_effect" => decision(request, false),
        "decide_subject_bound" => decision(request, true),
        "combine_decisions" => combine(request),
        _ => Err(err("unknown judge statistic")),
    }
}

pub(crate) fn terminating_string(v: &Rat) -> String {
    if v.is_zero() {
        return "0".into();
    }
    let mut exponent = -1000;
    let mut n = (v / ten(exponent)).to_integer();
    while (&n % 10u8).is_zero() {
        n /= 10;
        exponent += 1
    }
    decimal_string(v, exponent)
}
fn width_bounds(
    count: usize,
    lower: &Rat,
    upper: &Rat,
    alpha: &Rat,
    alpha_text: &str,
    comparisons: &BigInt,
) -> Result<(Rat, Rat), AnalysisError> {
    let midpoint = (lower + upper) / integer(2);
    let mut widths = Vec::new();
    for mean in [lower.clone(), upper.clone(), midpoint] {
        let result = interval(
            &vec![mean; count],
            lower,
            upper,
            alpha,
            alpha_text,
            comparisons,
            -15,
        )?;
        widths.push(result.upper - result.lower)
    }
    let padding = ten(-15) * integer(2);
    let support = quantize(upper, -15, Rounding::Ceiling) - quantize(lower, -15, Rounding::Floor);
    Ok((
        (widths[0].clone().min(widths[1].clone()) - &padding).max(Rat::zero()),
        support.min(&widths[2] + padding),
    ))
}
fn precision_interval(
    count: usize,
    lower: &Rat,
    upper: &Rat,
    policy: &Value,
) -> Result<Value, AnalysisError> {
    let alpha = dec(&policy["alpha"], "alpha")?;
    let comparisons = positive_big(&policy["comparison_family_size"], "comparison_family_size")?;
    let alpha_text = policy["alpha"].as_str().unwrap();
    let limit = dec(&policy["maximum_interval_width"], "maximum_interval_width")?;
    let mut cache = BTreeMap::<usize, (Rat, Rat)>::new();
    let mut bounds = |n| -> Result<(Rat, Rat), AnalysisError> {
        if let Some(v) = cache.get(&n) {
            return Ok(v.clone());
        }
        let v = width_bounds(n, lower, upper, &alpha, alpha_text, &comparisons)?;
        cache.insert(n, v.clone());
        Ok(v)
    };
    let (minimum, maximum) = bounds(count)?;
    let status = if maximum <= limit {
        "within_limit"
    } else if minimum > limit {
        "unattainable"
    } else {
        "not_guaranteed"
    };
    let mut needed = None;
    if bounds(10000)?.1 <= limit {
        let (mut left, mut right) = (1usize, 10000usize);
        while left < right {
            let middle = (left + right) / 2;
            if bounds(middle)?.1 <= limit {
                right = middle
            } else {
                left = middle + 1
            }
        }
        needed = Some(left)
    }
    Ok(
        json!({"minimum_width_lower_bound":terminating_string(&minimum),"maximum_width_upper_bound":terminating_string(&maximum),"status":status,"units_for_guaranteed_width":needed}),
    )
}
/// Advisory precision only; does not authenticate collected ratings or execution.
pub fn judge_plan_precision(plan: &Value, policy: &Value) -> Result<Value, AnalysisError> {
    let cases = plan["sampling"]["case_units"]
        .as_array()
        .ok_or_else(|| err("precision planning requires 1 to 10000 independent units"))?;
    let units = cases
        .iter()
        .filter_map(|c| c["unit_id"].as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if !(1..=10000).contains(&units) {
        return Err(err(
            "precision planning requires 1 to 10000 independent units",
        ));
    }
    let ratings = plan["scale"]["ratings"]
        .as_array()
        .ok_or_else(|| err("declared rating scale must not be empty"))?;
    let values = ratings
        .iter()
        .map(|r| dec(&r["value"], "rating"))
        .collect::<Result<Vec<_>, _>>()?;
    let lower = values
        .iter()
        .min()
        .ok_or_else(|| err("declared rating scale must not be empty"))?;
    let upper = values.iter().max().unwrap();
    let width = upper - lower;
    let effect = precision_interval(units, &(-&width), &width, policy)?;
    let mut subject = precision_interval(units, lower, upper, policy)?;
    subject["role"] = json!(if policy["subject_bound"].is_null() {
        "descriptive"
    } else {
        "decision"
    });
    Ok(
        json!({"method":METHOD,"independent_units":units,"minimum_units":policy["minimum_units"],"minimum_units_met":units>=positive(&policy["minimum_units"],"minimum_units")?,"maximum_interval_width":policy["maximum_interval_width"],"unit_search_limit":10000,"paired_effect":effect,"subject_score":subject,"scope":"complete_planned_schedule"}),
    )
}

/// Exact decimal display facts for an already replayed judge measurement result.
/// This does not infer or change the recorded decision.
pub fn judge_report_numbers(
    plan: &Value,
    measurements: &Value,
    policy: &Value,
    analysis: &Value,
) -> Result<Value, AnalysisError> {
    crate::validate_measurement_plan(plan)?;
    crate::validate_judge_analysis_policy(policy, plan)?;
    let mut baseline = Value::Null;
    if !analysis["subject_interval"].is_null() {
        let mut cases = BTreeMap::<String, Vec<Rat>>::new();
        for trial in measurements["trials"]
            .as_array()
            .ok_or_else(|| err("judge trials must be an array"))?
        {
            if trial["side"] == "baseline" {
                cases
                    .entry(
                        trial["case_id"]
                            .as_str()
                            .ok_or_else(|| err("case ID must be text"))?
                            .into(),
                    )
                    .or_default()
                    .push(dec(&trial["parse"]["value"], "baseline value")?);
            }
        }
        let mut units = BTreeMap::<String, Vec<Rat>>::new();
        for case in plan["sampling"]["case_units"].as_array().unwrap() {
            let id = case["case_id"].as_str().unwrap();
            let scores = cases
                .get(id)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| err("baseline reporting requires every planned case"))?;
            units
                .entry(case["unit_id"].as_str().unwrap().into())
                .or_default()
                .push(scores.iter().cloned().sum::<Rat>() / integer(scores.len()));
        }
        let mean = units
            .values()
            .map(|s| s.iter().cloned().sum::<Rat>() / integer(s.len()))
            .sum::<Rat>()
            / integer(units.len());
        baseline = json!(decimal_string(
            &quantize(&mean, -15, Rounding::HalfEven),
            -15
        ));
    }
    let mut widths = serde_json::Map::new();
    for key in ["effect_interval", "subject_interval"] {
        let v = &analysis[key];
        if !v.is_null() {
            let width = dec(&v["upper"], "upper")? - dec(&v["lower"], "lower")?;
            let exponent = ["upper", "lower"]
                .into_iter()
                .map(|k| {
                    let raw = v[k].as_str().unwrap();
                    let mut p = raw.split(['e', 'E']);
                    let mantissa = p.next().unwrap();
                    p.next().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0)
                        - mantissa.split('.').nth(1).map(str::len).unwrap_or(0) as i32
                })
                .min()
                .unwrap();
            widths.insert(key.into(),json!({"width":decimal_string(&width,exponent),"passed":width<=dec(&policy["maximum_interval_width"],"maximum_interval_width")?}));
        }
    }
    let confidence = (Rat::one() - dec(&policy["alpha"], "alpha")?) * integer(100);
    let digits = policy["alpha"]
        .as_str()
        .unwrap()
        .split('.')
        .nth(1)
        .map(str::len)
        .unwrap_or(0) as i32;
    let mut confidence = decimal_string(&confidence, -digits);
    if confidence.contains('.') && !confidence.contains('E') {
        confidence = confidence
            .trim_end_matches('0')
            .trim_end_matches('.')
            .into()
    }
    Ok(json!({"baseline_mean":baseline,"widths":widths,"confidence":confidence}))
}
