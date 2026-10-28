//! Fixed paired SHA-256 resampling, with no Python dependency.
//!
//! Binary64 operations deliberately retain the Python reference evaluation order.
//! Compile using ordinary Rust floating-point semantics; fast-math and fused
//! replacement expressions are outside this experiment's supported configuration.

#[cfg(test)]
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
    fsum_with_scratch(values, &mut Vec::with_capacity(32))
}

fn fsum_with_scratch(
    values: impl IntoIterator<Item = f64>,
    partials: &mut Vec<f64>,
) -> Result<f64, IntervalError> {
    partials.clear();
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
    let mut draws = Vec::with_capacity(count);
    indexes_into(seed, count, replicate, &mut draws);
    draws
}

fn indexes_into(seed: &[u8; 32], count: usize, replicate: u64, draws: &mut Vec<usize>) {
    // The protocol always hashes exactly 48 bytes, so padding is fixed. Use
    // sha2's existing compression backend (including runtime CPU selection).
    // This removes general-purpose buffering/finalization, not any hash rounds.
    let mut block = sha2::digest::generic_array::GenericArray::default();
    block[..32].copy_from_slice(seed);
    block[32..40].copy_from_slice(&replicate.to_be_bytes());
    block[48] = 0x80;
    block[56..].copy_from_slice(&384u64.to_be_bytes());
    draws.clear();
    draws.extend((0..count).map(|draw| {
        block[40..48].copy_from_slice(&(draw as u64).to_be_bytes());
        let mut state = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        sha2::compress256(&mut state, std::slice::from_ref(&block));
        (((state[0] as u64) << 32 | state[1] as u64) % count as u64) as usize
    }));
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
    partials: &mut Vec<f64>,
) -> Result<f64, IntervalError> {
    let count = baseline.len() as f64;
    if metric != "normalized_nll_per_utf8_byte" {
        return Ok(
            fsum_with_scratch(draws.iter().map(|&i| subject[i] - baseline[i]), partials)? / count
                * 100.0,
        );
    }
    let baseline_mean = fsum_with_scratch(draws.iter().map(|&i| baseline[i]), partials)? / count;
    if baseline_mean <= 0.0 {
        return Err(IntervalError::NonPositiveBaseline);
    }
    Ok((fsum_with_scratch(draws.iter().map(|&i| subject[i]), partials)? / count) / baseline_mean)
}

/// Unsorted replicates in reference evaluation order. The singleton shortcut
/// returns one value, because the reference performs no SHA-256 draws then.
pub fn resampling_distribution(
    metric: &str,
    baseline: &[f64],
    subject: &[f64],
    schedule_sha256: &str,
) -> Result<Vec<f64>, IntervalError> {
    let available = if baseline.len().saturating_mul(REPLICATES) < 2 * MIN_DRAWS_PER_WORKER {
        1
    } else {
        std::thread::available_parallelism().map_or(1, usize::from)
    };
    distribution_with_workers(
        metric,
        baseline,
        subject,
        schedule_sha256,
        REPLICATES,
        bootstrap_workers(baseline.len(), REPLICATES, available),
    )
}

// Bound both thread count and per-worker scratch memory. available_parallelism
// accounts for the operating system's affinity and supported cgroup quotas.
const MAX_BOOTSTRAP_WORKERS: usize = 8;
// Release microcases on the development host amortize scoped-thread startup
// by about 4,096 draws per worker; workloads below 8,192 total draws stay serial. This deliberately
// keeps sub-millisecond cases serial even where extra threads can help.
const MIN_DRAWS_PER_WORKER: usize = 4_096;

fn bootstrap_workers(count: usize, replicates: usize, available: usize) -> usize {
    available
        .min(MAX_BOOTSTRAP_WORKERS)
        .min(replicates)
        .min(count.saturating_mul(replicates) / MIN_DRAWS_PER_WORKER)
        .max(1)
}

fn distribution_with_workers(
    metric: &str,
    baseline: &[f64],
    subject: &[f64],
    schedule_sha256: &str,
    replicates: usize,
    workers: usize,
) -> Result<Vec<f64>, IntervalError> {
    distribution_with_spawn_policy(
        metric,
        baseline,
        subject,
        schedule_sha256,
        replicates,
        workers,
        |_| Ok(()),
    )
}

fn distribution_with_spawn_policy(
    metric: &str,
    baseline: &[f64],
    subject: &[f64],
    schedule_sha256: &str,
    replicates: usize,
    workers: usize,
    mut permit_spawn: impl FnMut(usize) -> std::io::Result<()>,
) -> Result<Vec<f64>, IntervalError> {
    let seed = validate(baseline, subject, schedule_sha256)?;
    if baseline.len() == 1 {
        return Ok(vec![comparison(
            metric,
            baseline,
            subject,
            &[0],
            &mut Vec::with_capacity(32),
        )?]);
    }
    let workers = workers
        .clamp(1, MAX_BOOTSTRAP_WORKERS)
        .min(replicates.max(1));
    let compute = |first: usize, length: usize| {
        let mut output = Vec::with_capacity(length);
        let mut draws = Vec::with_capacity(baseline.len());
        let mut partials = Vec::with_capacity(32);
        for replicate in first..first + length {
            indexes_into(&seed, baseline.len(), replicate as u64, &mut draws);
            output.push(comparison(
                metric,
                baseline,
                subject,
                &draws,
                &mut partials,
            )?);
        }
        Ok::<_, IntervalError>(output)
    };
    if workers == 1 {
        return compute(0, replicates);
    }
    enum Chunk<'scope> {
        Spawned(std::thread::ScopedJoinHandle<'scope, Result<Vec<f64>, IntervalError>>),
        Inline(Result<Vec<f64>, IntervalError>),
    }
    let chunk_size = replicates.div_ceil(workers);
    std::thread::scope(|scope| {
        let chunks: Vec<_> = (0..replicates)
            .step_by(chunk_size)
            .enumerate()
            .map(|(chunk, first)| {
                let length = chunk_size.min(replicates - first);
                match permit_spawn(chunk).and_then(|()| {
                    std::thread::Builder::new().spawn_scoped(scope, move || compute(first, length))
                }) {
                    Ok(handle) => Chunk::Spawned(handle),
                    // Thread creation is an optional optimization. If the OS
                    // cannot create a worker, compute that range on this thread.
                    Err(_) => Chunk::Inline(compute(first, length)),
                }
            })
            .collect();
        let mut distribution = Vec::with_capacity(replicates);
        // Keep inline results in range order too: a later range's error must
        // never displace an earlier worker's analytical error. Every replicate
        // retains the same serial draws and arithmetic regardless of placement.
        for chunk in chunks {
            let output = match chunk {
                Chunk::Spawned(handle) => handle.join().expect("bootstrap worker panicked"),
                Chunk::Inline(result) => result,
            }?;
            distribution.extend(output);
        }
        Ok(distribution)
    })
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
    check_policy, check_run, compare_runs, compare_validated_run_views, compare_validated_runs,
    validate_policy, validate_run, verify_input_projection, AnalysisError, ValidatedPolicy,
    ValidatedRun, ValidatedRunView, DEFAULT_MAX_BOOTSTRAP_DRAWS,
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
    render_judge_runtime_prompt, validate_frozen_answers, validate_judge_analysis_policy,
    validate_measurement_plan, validate_measurements,
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

#[cfg(test)]
mod bootstrap_tests {
    use super::*;

    const SEED: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    // Independent high-level digest oracle retains the original protocol.
    fn reference_indexes(seed: &[u8; 32], count: usize, replicate: u64) -> Vec<usize> {
        let mut input = [0u8; 48];
        input[..32].copy_from_slice(seed);
        input[32..40].copy_from_slice(&replicate.to_be_bytes());
        (0..count)
            .map(|draw| {
                input[40..].copy_from_slice(&(draw as u64).to_be_bytes());
                let digest = Sha256::digest(input);
                (u64::from_be_bytes(digest[..8].try_into().unwrap()) % count as u64) as usize
            })
            .collect()
    }

    #[test]
    fn fixed_block_draws_match_independent_digest() {
        for seed in [[0; 32], [255; 32], std::array::from_fn(|i| i as u8)] {
            for count in [1, 2, 3, 17, 257, 4097] {
                for replicate in [0, 1, 2047, u32::MAX as u64, u64::MAX] {
                    assert_eq!(
                        indexes(&seed, count, replicate),
                        reference_indexes(&seed, count, replicate)
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "bounded timing diagnostic; run explicitly in release mode"]
    fn fixed_block_microcases() {
        for count in [64, 1024, 50_000] {
            for round in 0..5 {
                for fixed in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = std::time::Instant::now();
                    for replicate in 0..32 {
                        let result = if fixed {
                            indexes(&[0xa5; 32], count, replicate)
                        } else {
                            reference_indexes(&[0xa5; 32], count, replicate)
                        };
                        std::hint::black_box(result);
                    }
                    eprintln!(
                        "count={count} round={round} fixed={fixed} seconds={:.9}",
                        start.elapsed().as_secs_f64()
                    );
                }
            }
        }
    }

    // Keep the allocation-per-replicate reference to check scratch reuse as well
    // as scheduling. This deliberately follows the previous serial implementation.
    fn serial_reference(
        metric: &str,
        baseline: &[f64],
        subject: &[f64],
        replicates: usize,
    ) -> Result<Vec<f64>, IntervalError> {
        let seed = validate(baseline, subject, SEED)?;
        (0..replicates)
            .map(|replicate| {
                let draws = reference_indexes(&seed, baseline.len(), replicate as u64);
                let count = baseline.len() as f64;
                if metric != "normalized_nll_per_utf8_byte" {
                    return Ok(
                        fsum(draws.iter().map(|&i| subject[i] - baseline[i]))? / count * 100.0,
                    );
                }
                let baseline_mean = fsum(draws.iter().map(|&i| baseline[i]))? / count;
                if baseline_mean <= 0.0 {
                    return Err(IntervalError::NonPositiveBaseline);
                }
                Ok((fsum(draws.iter().map(|&i| subject[i]))? / count) / baseline_mean)
            })
            .collect()
    }

    fn bits(result: Result<Vec<f64>, IntervalError>) -> Result<Vec<u64>, IntervalError> {
        result.map(|values| values.into_iter().map(f64::to_bits).collect())
    }

    #[test]
    fn worker_counts_preserve_every_replicate_bit() {
        let fixtures = [
            ("custom", vec![0.0; 5], vec![1e-16, 1.0, 1e16, -1e16, -1.0]),
            ("custom", vec![0.0, -0.0, 0.0], vec![-0.0, 0.0, -0.0]),
            (
                "custom",
                vec![0.0; 3],
                vec![f64::from_bits(1), -f64::from_bits(1), 0.0],
            ),
            (
                "normalized_nll_per_utf8_byte",
                vec![0.125, 1.0, 3.0, 9.0],
                vec![0.2, 0.75, 4.0, 5.0],
            ),
            (
                "normalized_nll_per_utf8_byte",
                vec![f64::from_bits(1); 2],
                vec![1.0; 2],
            ),
            ("custom", vec![-f64::MAX; 2], vec![f64::MAX; 2]),
        ];
        for (metric, baseline, subject) in fixtures {
            for replicates in [1, 3, 17, 65, REPLICATES] {
                let expected = bits(serial_reference(metric, &baseline, &subject, replicates));
                for workers in [1, 2, 4, 8] {
                    assert_eq!(
                        bits(distribution_with_workers(
                            metric, &baseline, &subject, SEED, replicates, workers
                        )),
                        expected,
                        "{metric}, {replicates} replicates, {workers} workers"
                    );
                }
            }
        }
    }

    #[test]
    fn worker_counts_preserve_serial_errors_and_validation_order() {
        let fixtures = [
            (
                "normalized_nll_per_utf8_byte",
                vec![1e308, 1e308, -1e308],
                vec![1.0; 3],
            ),
            ("custom", vec![0.0; 3], vec![1e308, 1e308, -1e308]),
            (
                "custom",
                vec![-f64::MAX, f64::MAX],
                vec![f64::MAX, -f64::MAX],
            ),
            (
                "normalized_nll_per_utf8_byte",
                vec![-1.0, 0.0, 1.0],
                vec![1.0; 3],
            ),
        ];
        // This fixture has different errors across replicate ranges, so a worker
        // that finishes first cannot be allowed to choose the returned error.
        let seed = seed_bytes(SEED).unwrap();
        let errors: Vec<_> = (0..65)
            .filter_map(|replicate| {
                comparison(
                    fixtures[0].0,
                    &fixtures[0].1,
                    &fixtures[0].2,
                    &indexes(&seed, 3, replicate),
                    &mut Vec::new(),
                )
                .err()
            })
            .collect();
        assert!(errors.contains(&IntervalError::IntermediateOverflow));
        assert!(errors.contains(&IntervalError::NonPositiveBaseline));
        for (metric, baseline, subject) in fixtures {
            let expected = bits(serial_reference(metric, &baseline, &subject, 65));
            assert!(expected.is_err());
            for workers in [1, 2, 4, 8] {
                assert_eq!(
                    bits(distribution_with_workers(
                        metric, &baseline, &subject, SEED, 65, workers
                    )),
                    expected
                );
            }
        }
        for workers in [1, 2, 4, 8] {
            assert_eq!(
                distribution_with_workers("custom", &[], &[], "invalid", 65, workers),
                Err(IntervalError::InvalidSeed)
            );
            assert_eq!(
                distribution_with_workers("custom", &[f64::NAN], &[1.0], "invalid", 65, workers),
                Err(IntervalError::InvalidSeed)
            );
            assert_eq!(
                distribution_with_workers("custom", &[1.0], &[], SEED, 65, workers),
                Err(IntervalError::InvalidLength)
            );
            assert_eq!(
                distribution_with_workers("custom", &[f64::NAN], &[1.0], SEED, 65, workers),
                Err(IntervalError::NonFiniteInput)
            );
            assert_eq!(
                distribution_with_workers("custom", &[0.25], &[0.5], SEED, 65, workers).unwrap(),
                vec![25.0]
            );
        }
    }

    #[test]
    fn failed_worker_creation_preserves_bits_and_first_analytical_error() {
        let fixtures = [
            ("custom", vec![0.0; 5], vec![1e-16, 1.0, 1e16, -1e16, -1.0]),
            (
                "custom",
                vec![0.0; 3],
                vec![f64::from_bits(1), -f64::from_bits(1), 0.0],
            ),
            (
                "normalized_nll_per_utf8_byte",
                vec![0.125, 1.0, 3.0, 9.0],
                vec![0.2, 0.75, 4.0, 5.0],
            ),
            (
                "normalized_nll_per_utf8_byte",
                vec![1e308, 1e308, -1e308],
                vec![1.0; 3],
            ),
            ("custom", vec![0.0; 3], vec![1e308, 1e308, -1e308]),
            (
                "custom",
                vec![-f64::MAX, f64::MAX],
                vec![f64::MAX, -f64::MAX],
            ),
        ];
        for (metric, baseline, subject) in fixtures {
            for replicates in [3, 17, 65, REPLICATES] {
                let expected = bits(serial_reference(metric, &baseline, &subject, replicates));
                for workers in [2, 4, 8] {
                    let workers = workers.min(replicates);
                    let chunk_size = replicates.div_ceil(workers);
                    let chunk_count = replicates.div_ceil(chunk_size);
                    // Fail the first attempt, a middle attempt, or every attempt.
                    // The policy feeds the same Result branch as spawn_scoped's
                    // OS failure, without disrupting the process's actual limits.
                    for failure in [Some(0), Some(chunk_count / 2), None] {
                        let mut attempts = 0;
                        let actual = distribution_with_spawn_policy(
                            metric,
                            &baseline,
                            &subject,
                            SEED,
                            replicates,
                            workers,
                            |chunk| {
                                attempts += 1;
                                if failure.is_none() || failure == Some(chunk) {
                                    Err(std::io::Error::new(
                                        std::io::ErrorKind::WouldBlock,
                                        "injected worker creation failure",
                                    ))
                                } else {
                                    Ok(())
                                }
                            },
                        );
                        assert_eq!(attempts, chunk_count);
                        assert_eq!(bits(actual), expected, "{metric}, {replicates} replicates, {workers} workers, failure {failure:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn scratch_resets_after_overflow_and_partial_rounding() {
        let mut partials = Vec::new();
        assert_eq!(
            fsum_with_scratch([1e308, 1e308], &mut partials),
            Err(IntervalError::IntermediateOverflow)
        );
        for values in [[1e16, 1.0, -1e16], [1e-16, 1.0, 1e16], [-0.0, 0.0, -0.0]] {
            assert_eq!(
                fsum_with_scratch(values, &mut partials).unwrap().to_bits(),
                fsum(values).unwrap().to_bits()
            );
        }
    }

    #[test]
    fn worker_selection_bounds_threads_and_small_work() {
        assert_eq!(bootstrap_workers(2, REPLICATES, 64), 1);
        assert_eq!(
            bootstrap_workers(50_000, REPLICATES, 64),
            MAX_BOOTSTRAP_WORKERS
        );
        assert_eq!(bootstrap_workers(50_000, REPLICATES, 2), 2);
        assert_eq!(bootstrap_workers(50_000, 3, 64), 3);
        assert_eq!(
            bootstrap_workers(usize::MAX, REPLICATES, 64),
            MAX_BOOTSTRAP_WORKERS
        );
    }

    #[test]
    #[ignore = "bounded timing diagnostic; run explicitly in release mode"]
    fn bootstrap_worker_microcases() {
        for count in [2, 8, 16, 32, 64, 256, 1024] {
            let baseline = vec![1.0; count];
            let subject = vec![1.25; count];
            for workers in [1, 2, 4, 8] {
                let mut times = Vec::new();
                for _ in 0..3 {
                    let started = std::time::Instant::now();
                    std::hint::black_box(
                        distribution_with_workers(
                            "custom", &baseline, &subject, SEED, REPLICATES, workers,
                        )
                        .unwrap(),
                    );
                    times.push(started.elapsed().as_secs_f64() * 1000.0);
                }
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "draws={} workers={} median_ms={:.3}",
                    count * REPLICATES,
                    workers,
                    times[1]
                );
            }
        }
    }
}
