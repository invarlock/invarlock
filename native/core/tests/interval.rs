use invarlock_native_core::{fsum, paired_resampling_interval, sample_indexes, IntervalError};

#[test]
fn summation_retains_cancellation_and_final_partial_rounding() {
    assert_eq!(fsum([1e16, 1.0, -1e16]).unwrap(), 1.0);
    assert_eq!(
        fsum([1e-16, 1.0, 1e16]).unwrap().to_bits(),
        (1e16_f64 + 2.0).to_bits()
    );
    assert_eq!(
        fsum([f64::from_bits(1), f64::from_bits(1)])
            .unwrap()
            .to_bits(),
        2
    );
    assert_eq!(fsum([-0.0]).unwrap().to_bits(), 0);
}

#[test]
fn summation_does_not_hide_intermediate_overflow() {
    assert_eq!(
        fsum([1e308, 1e308, -1e308]),
        Err(IntervalError::IntermediateOverflow)
    );
}

#[test]
fn one_pair_and_input_errors() {
    let seed = "a".repeat(64);
    assert_eq!(
        paired_resampling_interval("normalized_nll_per_utf8_byte", &[2.0], &[3.0], &seed).unwrap(),
        (1.5, 1.5)
    );
    assert_eq!(
        paired_resampling_interval("custom", &[0.25], &[0.5], &seed).unwrap(),
        (25.0, 25.0)
    );
    assert_eq!(
        paired_resampling_interval("custom", &[], &[], &seed),
        Err(IntervalError::InvalidLength)
    );
    assert_eq!(
        paired_resampling_interval("custom", &[0.0], &[], &seed),
        Err(IntervalError::InvalidLength)
    );
    assert_eq!(
        paired_resampling_interval("custom", &[0.0], &[1.0], &"A".repeat(64)),
        Err(IntervalError::InvalidSeed)
    );
    assert_eq!(
        paired_resampling_interval("custom", &[f64::NAN], &[1.0], &seed),
        Err(IntervalError::NonFiniteInput)
    );
    assert_eq!(
        paired_resampling_interval("normalized_nll_per_utf8_byte", &[0.0], &[1.0], &seed),
        Err(IntervalError::NonPositiveBaseline)
    );
}

#[test]
fn draws_are_stable_and_in_range() {
    let draws = sample_indexes(&"a".repeat(64), 7, 3).unwrap();
    assert_eq!(draws, vec![2, 0, 5, 4, 1, 6, 5]);
    assert!(draws.iter().all(|&i| i < 7));
    assert_eq!(draws, sample_indexes(&"a".repeat(64), 7, 3).unwrap());
}

#[test]
fn full_nonconstant_intervals_match_python_golden_bits() {
    let seed = "a".repeat(64);
    let (lower, upper) = paired_resampling_interval(
        "normalized_nll_per_utf8_byte",
        &[0.125, 1.0, 3.0, 9.0],
        &[0.2, 0.75, 4.0, 5.0],
        &seed,
    )
    .unwrap();
    assert_eq!(lower.to_bits(), 0x3fe2000000000000);
    assert_eq!(upper.to_bits(), 0x3ff5810624dd2f1b);
    let (lower, upper) =
        paired_resampling_interval("custom", &[0.0, 0.0, 0.0], &[1e-16, 1.0, 1e16], &seed).unwrap();
    assert_eq!(lower.to_bits(), 0x3d06849b86a12b9b);
    assert_eq!(upper.to_bits(), 0x43abc16d674ec800);
}

#[test]
fn computed_infinity_and_nan_are_left_for_outer_report_validation() {
    let seed = "a".repeat(64);
    let (lower, upper) = paired_resampling_interval(
        "normalized_nll_per_utf8_byte",
        &[f64::from_bits(1)],
        &[1.0],
        &seed,
    )
    .unwrap();
    assert_eq!((lower, upper), (f64::INFINITY, f64::INFINITY));
    let (lower, upper) = paired_resampling_interval(
        "normalized_nll_per_utf8_byte",
        &[f64::from_bits(1); 2],
        &[1.0; 2],
        &seed,
    )
    .unwrap();
    assert!(lower.is_nan() && upper.is_nan());
    assert_eq!(
        fsum([f64::INFINITY, f64::NEG_INFINITY]),
        Err(IntervalError::OppositeInfinities)
    );
}
