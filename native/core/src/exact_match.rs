use crate::AnalysisError;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};
use serde_json::{json, Value};
const V1: &str = "newcombe_hybrid_score_paired_v1";
const V2: &str = "newcombe_hybrid_score_paired_v2";
fn binary(values: &Value, label: &str) -> Result<Vec<bool>, AnalysisError> {
    let values = values
        .as_array()
        .ok_or_else(|| AnalysisError::new(format!("{label} outcomes must be a sequence")))?;
    if values.is_empty() {
        return Err(AnalysisError::new(format!(
            "{label} outcomes must be non-empty"
        )));
    }
    if values.len() > 50000 {
        return Err(AnalysisError::new(format!(
            "{label} outcomes exceed the 50_000-pair limit"
        )));
    }
    values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if let Some(b) = v.as_bool() {
                Ok(b)
            } else if v.is_number() && matches!(v.as_f64(), Some(0.0) | Some(1.0)) {
                Ok(v.as_f64() == Some(1.0))
            } else {
                Err(AnalysisError::new(format!(
                    "{label} outcome {i} must be binary (zero or one)"
                )))
            }
        })
        .collect()
}
fn wilson(successes: usize, count: usize) -> (f64, f64) {
    let n = count as f64;
    let proportion = successes as f64 / n;
    let z = 1.959963984540054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (proportion + z2 / (2.0 * n)) / denominator;
    let radius =
        z * (proportion * (1.0 - proportion) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    ((center - radius).max(0.0), (center + radius).min(1.0))
}
pub fn paired_exact_match_statistics(
    baseline: &Value,
    subject: &Value,
    method: &str,
) -> Result<Value, AnalysisError> {
    if ![V1, V2].contains(&method) {
        return Err(AnalysisError::new(
            "confidence interval method is unsupported",
        ));
    }
    let b = binary(baseline, "baseline")?;
    let s = binary(subject, "subject")?;
    if b.len() != s.len() {
        return Err(AnalysisError::new(
            "baseline and subject outcomes must contain the same number of pairs",
        ));
    }
    let (mut both_pass, mut both_fail, mut lost, mut gained) = (0usize, 0usize, 0usize, 0usize);
    for (&b, &s) in b.iter().zip(&s) {
        match (b, s) {
            (true, true) => both_pass += 1,
            (false, false) => both_fail += 1,
            (true, false) => lost += 1,
            _ => gained += 1,
        }
    }
    let count = b.len();
    let n = count as f64;
    let bp = both_pass + lost;
    let sp = both_pass + gained;
    let br = bp as f64 / n;
    let sr = sp as f64 / n;
    let effect = sr - br;
    let (sl, su) = wilson(sp, count);
    let (bl, bu) = wilson(bp, count);
    let correlation = if method == V1 {
        let denominator = (sr * (1.0 - sr) * br * (1.0 - br)).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            ((both_pass as f64 / n - sr * br) / denominator).clamp(-1.0, 1.0)
        }
    } else {
        // Multiplication is exact integer arithmetic before conversion, as in Python.
        let denominator =
            ((bp as u128 * (gained + both_fail) as u128 * sp as u128 * (lost + both_fail) as u128)
                as f64)
                .sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            let raw = (both_pass as i64 * both_fail as i64 - lost as i64 * gained as i64) as f64;
            let correction = n / 2.0;
            let numerator = if raw > 0.0 {
                (raw - correction).max(0.0)
            } else if raw < 0.0 {
                (raw + correction).min(0.0)
            } else {
                raw
            };
            (numerator / denominator).clamp(-1.0, 1.0)
        }
    };
    let a = sr - sl;
    let b = bu - br;
    let lower_rad = a.powi(2) + b.powi(2) - 2.0 * correlation * a * b;
    let a = su - sr;
    let b = br - bl;
    let upper_rad = a.powi(2) + b.powi(2) - 2.0 * correlation * a * b;
    let (lower, upper) = if method == V1 && lost == 0 && gained == 0 {
        (0.0, 0.0)
    } else {
        (
            (effect - lower_rad.max(0.0).sqrt()).max(-1.0) * 100.0,
            (effect + upper_rad.max(0.0).sqrt()).min(1.0) * 100.0,
        )
    };
    let discordant = lost + gained;
    let probability = if discordant == 0 {
        1.0
    } else {
        let mut coefficient = BigInt::one();
        let mut cumulative = BigInt::one();
        for successes in 1..=lost.min(gained) {
            coefficient = coefficient * (discordant - successes + 1) / successes;
            cumulative += &coefficient
        }
        BigRational::new(cumulative * 2, BigInt::one() << discordant)
            .to_f64()
            .unwrap_or(0.0)
            .min(1.0)
    };
    Ok(
        json!({"pair_count":count,"baseline_pass_count":bp,"subject_pass_count":sp,"baseline_pass_subject_fail_count":lost,"baseline_fail_subject_pass_count":gained,"both_pass_count":both_pass,"both_fail_count":both_fail,"mcnemar_exact_two_sided_p_value":probability,"effect_size_pp":(gained as f64-lost as f64)/n*100.0,"effect_size_confidence_interval":{"method":method,"confidence_level":0.95,"lower_pp":lower,"upper_pp":upper}}),
    )
}
