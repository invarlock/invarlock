//! Fixed paired SHA-256 resampling, with no Python dependency.
//!
//! Binary64 operations deliberately retain the Python reference evaluation order.
//! Compile using ordinary Rust floating-point semantics; fast-math and fused
//! replacement expressions are outside this experiment's supported configuration.

use sha2::{Digest, Sha256};
use std::fmt;

pub const REPLICATES: usize = 2_048;
pub const CONFIDENCE: f64 = 0.95;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntervalError {
    InvalidLength,
    InvalidSeed,
    NonFiniteInput,
    IntermediateOverflow,
    OppositeInfinities,
    NonPositiveBaseline,
}

impl fmt::Display for IntervalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLength => "paired scores must have equal nonzero lengths",
            Self::InvalidSeed => "paired records schedule_sha256 is invalid",
            Self::NonFiniteInput => "paired scores must contain finite numbers",
            Self::IntermediateOverflow => "intermediate overflow in fsum",
            Self::OppositeInfinities => "-inf + inf in fsum",
            Self::NonPositiveBaseline => "normalized NLL baseline mean must be greater than zero",
        })
    }
}
impl std::error::Error for IntervalError {}

/// Accurate sum adapted from CPython 3.12.13 math_fsum.
/// See THIRD_PARTY_NOTICES.md and licenses/CPython-LICENSE.txt.
/// Includes CPython's final multiple-partial half-even correction, intermediate
/// overflow rejection, special values, and positive zero for an empty/zero sum.
pub fn fsum(values: impl IntoIterator<Item = f64>) -> Result<f64, IntervalError> {
    let mut partials: Vec<f64> = Vec::with_capacity(32);
    let mut special_sum = 0.0;
    let mut inf_sum = 0.0;
    for mut x in values {
        let original = x;
        let mut retained = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            let lo = y - yr;
            if lo != 0.0 {
                partials[retained] = lo;
                retained += 1;
            }
            x = hi;
        }
        partials.truncate(retained);
        if x != 0.0 {
            if !x.is_finite() {
                if original.is_finite() {
                    return Err(IntervalError::IntermediateOverflow);
                }
                if original.is_infinite() {
                    inf_sum += original;
                }
                special_sum += original;
                partials.clear();
            } else {
                partials.push(x);
            }
        }
    }
    if special_sum != 0.0 {
        if inf_sum.is_nan() {
            return Err(IntervalError::OppositeInfinities);
        }
        return Ok(special_sum);
    }
    let mut hi = 0.0;
    let mut lo = 0.0;
    if let Some(last) = partials.pop() {
        hi = last;
        while let Some(y) = partials.pop() {
            let x = hi;
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if let Some(&next) = partials.last() {
            if (lo < 0.0 && next < 0.0) || (lo > 0.0 && next > 0.0) {
                let y = lo * 2.0;
                let x = hi + y;
                let yr = x - hi;
                if y == yr {
                    hi = x;
                }
            }
        }
    }
    Ok(hi)
}

fn seed_bytes(schedule_sha256: &str) -> Result<[u8; 32], IntervalError> {
    let text = schedule_sha256.as_bytes();
    if text.len() != 64
        || !text
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
    {
        return Err(IntervalError::InvalidSeed);
    }
    let mut seed = [0; 32];
    for (i, pair) in text.chunks_exact(2).enumerate() {
        let nibble = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        seed[i] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(seed)
}

fn indexes(seed: &[u8; 32], count: usize, replicate: u64) -> Vec<usize> {
    let mut input = [0u8; 48];
    input[..32].copy_from_slice(seed);
    input[32..40].copy_from_slice(&replicate.to_be_bytes());
    (0..count)
        .map(|draw| {
            input[40..].copy_from_slice(&(draw as u64).to_be_bytes());
            let digest = Sha256::digest(input);
            (u64::from_be_bytes(digest[..8].try_into().expect("eight digest bytes")) % count as u64)
                as usize
        })
        .collect()
}

/// Diagnostic entry point for checking the complete SHA-256 draw protocol.
pub fn sample_indexes(
    schedule_sha256: &str,
    count: usize,
    replicate: u64,
) -> Result<Vec<usize>, IntervalError> {
    let seed = seed_bytes(schedule_sha256)?;
    if count == 0 {
        return Err(IntervalError::InvalidLength);
    }
    Ok(indexes(&seed, count, replicate))
}

fn validate(
    baseline: &[f64],
    subject: &[f64],
    schedule_sha256: &str,
) -> Result<[u8; 32], IntervalError> {
    // Match the reference's seed-first validation, including the singleton path.
    let seed = seed_bytes(schedule_sha256)?;
    if baseline.is_empty() || baseline.len() != subject.len() {
        return Err(IntervalError::InvalidLength);
    }
    if baseline.iter().chain(subject).any(|x| !x.is_finite()) {
        return Err(IntervalError::NonFiniteInput);
    }
    Ok(seed)
}

fn comparison(
    metric: &str,
    baseline: &[f64],
    subject: &[f64],
    draws: &[usize],
) -> Result<f64, IntervalError> {
    let count = baseline.len() as f64;
    if metric != "normalized_nll_per_utf8_byte" {
        return Ok(fsum(draws.iter().map(|&i| subject[i] - baseline[i]))? / count * 100.0);
    }
    let baseline_mean = fsum(draws.iter().map(|&i| baseline[i]))? / count;
    if baseline_mean <= 0.0 {
        return Err(IntervalError::NonPositiveBaseline);
    }
    Ok((fsum(draws.iter().map(|&i| subject[i]))? / count) / baseline_mean)
}

/// Unsorted replicates in reference evaluation order. The singleton shortcut
/// returns one value, because the reference performs no SHA-256 draws then.
pub fn resampling_distribution(
    metric: &str,
    baseline: &[f64],
    subject: &[f64],
    schedule_sha256: &str,
) -> Result<Vec<f64>, IntervalError> {
    let seed = validate(baseline, subject, schedule_sha256)?;
    if baseline.len() == 1 {
        return Ok(vec![comparison(metric, baseline, subject, &[0])?]);
    }
    let mut distribution = Vec::with_capacity(REPLICATES);
    for replicate in 0..REPLICATES {
        let draws = indexes(&seed, baseline.len(), replicate as u64);
        distribution.push(comparison(metric, baseline, subject, &draws)?);
    }
    Ok(distribution)
}

fn percentile(ordered: &[f64], quantile: f64) -> f64 {
    let position = (ordered.len() - 1) as f64 * quantile;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return ordered[lower];
    }
    let fraction = position - lower as f64;
    ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
}

/// Full fixed paired interval in one call. Supplied vectors must be nonempty,
/// equal-length, and finite. Computed infinity/NaN behavior matches the helper;
/// the caller owns report-level validation of computed values.
pub fn paired_resampling_interval(
    metric: &str,
    baseline: &[f64],
    subject: &[f64],
    schedule_sha256: &str,
) -> Result<(f64, f64), IntervalError> {
    let mut distribution = resampling_distribution(metric, baseline, subject, schedule_sha256)?;
    if distribution.len() == 1 {
        return Ok((distribution[0], distribution[0]));
    }
    // Stable numeric sorting matches Python for equal values, including signed
    // zeros. Finite supplied inputs cannot produce NaN replicates: fsum rejects
    // opposite infinities, and ratio denominators are strictly positive.
    distribution.sort_by(|a, b| a.partial_cmp(b).expect("replicates cannot be NaN"));
    let alpha = (1.0 - CONFIDENCE) / 2.0;
    Ok((
        percentile(&distribution, alpha),
        percentile(&distribution, 1.0 - alpha),
    ))
}

mod analysis;
mod exact_match;
mod idna;
mod idna_tables;
mod scoring;
mod unicode_tables;
pub use analysis::{
    check_policy, check_run, compare_runs, compare_validated_runs, validate_policy, validate_run,
    verify_input_projection, AnalysisError, ValidatedPolicy, ValidatedRun,
    DEFAULT_MAX_BOOTSTRAP_DRAWS,
};
pub use exact_match::paired_exact_match_statistics;
pub use scoring::{score, validate_configuration};
mod report;
pub use report::{build_comparison_report, validated_derived_measurements};
mod judge;
pub use judge::{judge_plan_precision, judge_statistics};
mod judge_contracts;
pub use judge_contracts::{
    expected_judge_trial_id, measurement_plan_digest, render_judge_request,
    render_judge_runtime_prompt, validate_judge_analysis_policy, validate_measurement_plan,
    validate_measurements,
};
mod judge_analysis;
pub use judge_analysis::{analyze_measurements, combine_judge_analysis_results};

mod judge_inspect;

mod judge_compatible;

mod judge_import;
mod judge_runtime;
pub use judge_import::{
    import_inspect_export, import_inspect_sample, validate_inspect_collection_budgets,
    validate_inspect_collection_options, InspectImportContext,
};
pub use judge_inspect::{replay_retained_inspect_source, validate_retained_inspect_event};

pub use judge_compatible::{
    normalize_openai_compatible_collection, normalized_compatible_base_url,
    openai_compatible_service_identity, validate_openai_compatible_collection,
};

pub use scoring::casefold_text as unicode_casefold;

pub use judge_compatible::render_compatible_judge_request;
pub use judge_contracts::validate_retained_collection_prefix;

pub use judge::judge_report_numbers;

mod preflight;
pub use preflight::captured_preflight_capacity;

pub use report::validate_scorer_binding;
